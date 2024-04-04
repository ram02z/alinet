import { Group, Text, rem } from '@mantine/core'
import { IconUpload, IconFileUpload, IconX } from '@tabler/icons-react'
import { Dropzone, MIME_TYPES } from '@mantine/dropzone'
import { useEffect } from 'react'
import classes from './DragDrop.module.css'
import cx from 'clsx'
import { FileWithId } from '../App'
import { v4 as uuidv4 } from 'uuid'

export interface DragDropProps {
  filesWithId: FileWithId[]
  setFilesWithId: any
}

export const DragDrop = ({ filesWithId, setFilesWithId }: DragDropProps) => {
  useEffect(() => {
    console.log(filesWithId)
  }, [filesWithId])

  return (
    <Dropzone
      className={cx(classes.dragdrop)}
      onDrop={(files: File[]) => {
        const filesWithId = files.map((file) => {
          return {
            id: uuidv4(),
            file: file,
          }
        })

        setFilesWithId((prevFiles: FileWithId[]) => [
          ...prevFiles,
          ...filesWithId,
        ])
      }}
      onReject={(files) => console.log('rejected files', files)}
      accept={[MIME_TYPES.mp4, MIME_TYPES.pdf]}
    >
      <Group
        className={cx(classes.dragdrop_content)}
        style={{ minHeight: rem(300), pointerEvents: 'none' }}
      >
        <Dropzone.Accept>
          <IconUpload
            size="3.2rem"
            stroke={1.5}
          />
        </Dropzone.Accept>
        <Dropzone.Reject>
          <IconX
            size="3.2rem"
            stroke={1.5}
          />
        </Dropzone.Reject>
        <Dropzone.Idle>
          <IconFileUpload
            size="3.2rem"
            stroke={1.5}
          />
        </Dropzone.Idle>

        <div>
          <Text
            size="xl"
            inline
          >
            Drag and drop here or click to select files
          </Text>
          <Text
            size="sm"
            color="dimmed"
            inline
            mt={7}
          >
            Attach as many lecture videos or supplementary pdfs as you like
          </Text>
        </div>
      </Group>
    </Dropzone>
  )
}
