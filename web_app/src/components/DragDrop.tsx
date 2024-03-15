import { Group, Text, rem } from '@mantine/core'
import { IconUpload, IconPhoto, IconX } from '@tabler/icons-react'
import { Dropzone, MIME_TYPES } from '@mantine/dropzone'
import { useEffect } from 'react'
import './DragDrop.css'

export interface DragDropProps {
  files: File[]
  setFiles: any
}

export const DragDrop = ({ files, setFiles }: DragDropProps) => {
  useEffect(() => {
    console.log(files)
  }, [files])

  return (
    <Dropzone
      className="dropzone"
      onDrop={(files: File[]) => {
        setFiles((prevFiles: File[]) => [...prevFiles, ...files])
      }}
      onReject={(files) => console.log('rejected files', files)}
      accept={[MIME_TYPES.pdf, MIME_TYPES.mp4]}
    >
      <Group
        className="dropzone-content"
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
          <IconPhoto
            size="3.2rem"
            stroke={1.5}
          />
        </Dropzone.Idle>

        <div>
          <Text
            size="xl"
            inline
          >
            Drag images here or click to select files
          </Text>
          <Text
            size="sm"
            color="dimmed"
            inline
            mt={7}
          >
            Attach as many files as you like, each file should not exceed 5mb
          </Text>
        </div>
      </Group>
    </Dropzone>
  )
}
