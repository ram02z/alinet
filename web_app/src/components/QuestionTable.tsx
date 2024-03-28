import cx from "clsx";
import { Table, Checkbox, rem } from "@mantine/core";
import classes from "./QuestionTable.module.css";
import { Question } from "../App";

export interface QuestionTableProps {
  selection: string[];
  setSelection: any;
  questions: Question[];
}

export const QuestionTable = ({
  selection,
  setSelection,
  questions,
}: QuestionTableProps) => {
  const toggleRow = (id: string) => {
    setSelection((current: string[]) =>
      current.includes(id)
        ? current.filter((item: string) => item !== id)
        : [...current, id],
    );
  };

  const toggleAll = () => {
    setSelection((current: string[]) =>
      current.length === questions.length
        ? []
        : questions.map((item) => item.id),
    );
  };

  const rows = questions.map((item: Question) => {
    const selected = selection.includes(item.id);
    return (
      <Table.Tr
        key={item.id}
        className={cx({ [classes.rowSelected]: selected })}
      >
        <Table.Td>
          <Checkbox
            checked={selection.includes(item.id)}
            onChange={() => toggleRow(item.id)}
          />
        </Table.Td>
        <Table.Td>{item.text}</Table.Td>
      </Table.Tr>
    );
  });

  return (
    <Table verticalSpacing="sm">
      <Table.Thead>
        <Table.Tr>
          <Table.Th style={{ width: rem(40) }}>
            <Checkbox
              onChange={toggleAll}
              checked={
                questions.length > 0 && selection.length === questions.length
              }
              indeterminate={
                selection.length > 0 && selection.length !== questions.length
              }
            />
          </Table.Th>
          <Table.Th>Questions</Table.Th>
        </Table.Tr>
      </Table.Thead>

      <Table.Tbody>{rows}</Table.Tbody>
    </Table>
  );
};
